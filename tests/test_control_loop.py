"""Tests for ClosedLoopController — step(), run(), invariants."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from eie.boundary.boundary import Boundary
from eie.control.gate import SafetyGate
from eie.control.loop import ClosedLoopController, LoopConfig, LoopIteration
from eie.control.policy import BatteryPolicy, ThermalStoragePolicy
from eie.core.enums import BoundaryType, MeasurementMethod
from eie.flows.base import Metadata
from eie.flows.storage import BatteryState, ThermalLayer, ThermalStorageState
from eie.optimization.optimizer import ShadowOptimizer
from eie.optimization.snapshot import BatteryConstraints, GridConstraints, SiteSnapshot, ThermalStorageConstraints
from eie.reference.state import ReferenceState


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)

@pytest.fixture
def reference(now: datetime) -> ReferenceState:
    return ReferenceState(reference_state_id="ref-loop-test", timestamp=now,
        ambient_temperature_k=293.15, ambient_pressure_pa=101_325.0, confidence=0.99, valid_until=now + timedelta(hours=2))

@pytest.fixture
def boundary(reference: ReferenceState, now: datetime) -> Boundary:
    return Boundary(boundary_id="bnd-loop-test", boundary_type=BoundaryType.SITE,
        included_entity_ids=["battery", "ts"], excluded_entity_ids=[],
        reference_state_id=reference.reference_state_id,
        accounting_period_start=now, accounting_period_end=now + timedelta(hours=2))

@pytest.fixture
def meta(reference: ReferenceState, boundary: Boundary, now: datetime) -> Metadata:
    return Metadata(timestamp=now, source="loop-test", method=MeasurementMethod.SIMULATED, unit="J",
        confidence=1.0, uncertainty=0.0, boundary_id=boundary.boundary_id, reference_state_id=reference.reference_state_id)

@pytest.fixture
def snapshot(reference: ReferenceState, boundary: Boundary, meta: Metadata, now: datetime) -> SiteSnapshot:
    battery_state = BatteryState(storage_id="b1", stored_energy_j=3_600_000.0*5, soc=0.60, soh=0.95,
        reserve_energy_j=3_600_000.0*0.5, boundary_id=boundary.boundary_id, reference_state_id=reference.reference_state_id, metadata=meta)
    ts_state = ThermalStorageState(storage_id="ts1", layers=[ThermalLayer(temperature_k=333.15, energy_j=2_000_000.0)],
        boundary_id=boundary.boundary_id, reference_state_id=reference.reference_state_id, metadata=meta)
    return SiteSnapshot(snapshot_id="snap-loop-001", timestamp=now, reference_state=reference, boundary=boundary,
        horizon_duration_s=3600.0, available_pv_power_w=6_000.0, battery_state=battery_state,
        battery_constraints=BatteryConstraints(max_charge_power_w=3_000.0, max_discharge_power_w=3_000.0,
            min_soc=0.10, max_soc=0.95, round_trip_efficiency=0.92, capacity_j=3_600_000.0*10.0),
        thermal_storage_state=ts_state,
        thermal_storage_constraints=ThermalStorageConstraints(max_charge_power_w=2_000.0, max_discharge_power_w=2_000.0,
            charge_efficiency=0.95, discharge_efficiency=0.95, standby_loss_w=50.0),
        building_electric_load_w=1_200.0, heat_pump_rated_cop=3.5, heat_pump_max_power_w=2_000.0,
        delivered_heat_temperature_k=318.15, heat_demand_w=3_000.0,
        grid_constraints=GridConstraints(import_limit_w=10_000.0, export_limit_w=5_000.0,
            import_tariff_per_kwh=0.28, export_tariff_per_kwh=0.05, carbon_intensity_kg_per_kwh=0.42))

@pytest.fixture
def controller(snapshot: SiteSnapshot) -> ClosedLoopController:
    config = LoopConfig(optimizer=ShadowOptimizer(grid_points_per_variable=5),
        policies=[BatteryPolicy(protected_reserve_fraction=0.20), ThermalStoragePolicy(min_comfort_k=293.15, max_comfort_k=340.0)],
        gate=SafetyGate())
    return ClosedLoopController(config)

def test_step_returns_loop_iteration(controller: ClosedLoopController, snapshot: SiteSnapshot, now: datetime) -> None:
    assert isinstance(controller.step(snapshot, at=now), LoopIteration)

def test_guard_bypass_count_always_zero(controller: ClosedLoopController, snapshot: SiteSnapshot, now: datetime) -> None:
    assert controller.step(snapshot, at=now).guard_bypass_count == 0

def test_approved_actions_are_gate_verified(controller: ClosedLoopController, snapshot: SiteSnapshot, now: datetime) -> None:
    iteration = controller.step(snapshot, at=now)
    approved_ids = {a.action_id for a in iteration.approved_actions}
    for verdict in iteration.gate_verdicts:
        if verdict.action_id in approved_ids:
            assert verdict.approved

def test_rejected_actions_not_approved(controller: ClosedLoopController, snapshot: SiteSnapshot, now: datetime) -> None:
    iteration = controller.step(snapshot, at=now)
    rejected_ids = {a.action_id for a in iteration.rejected_actions}
    for verdict in iteration.gate_verdicts:
        if verdict.action_id in rejected_ids:
            assert not verdict.approved

def test_exergy_improvement_non_negative(controller: ClosedLoopController, snapshot: SiteSnapshot, now: datetime) -> None:
    assert controller.step(snapshot, at=now).exergy_improvement_j >= 0.0

def test_snapshot_id_recorded(controller: ClosedLoopController, snapshot: SiteSnapshot, now: datetime) -> None:
    assert controller.step(snapshot, at=now).snapshot_id == snapshot.snapshot_id

def test_iteration_id_unique(controller: ClosedLoopController, snapshot: SiteSnapshot, now: datetime) -> None:
    assert controller.step(snapshot, at=now).iteration_id != controller.step(snapshot, at=now).iteration_id

def test_loop_iteration_frozen(controller: ClosedLoopController, snapshot: SiteSnapshot, now: datetime) -> None:
    with pytest.raises((AttributeError, TypeError)):
        controller.step(snapshot, at=now).guard_bypass_count = 1  # type: ignore[misc]

def test_loop_iteration_bypass_invariant() -> None:
    with pytest.raises(ValueError, match="guard_bypass_count"):
        LoopIteration(iteration_id="x", timestamp=datetime.now(timezone.utc), snapshot_id="s",
            optimization_result=None, proposed_actions=[], approved_actions=[], rejected_actions=[],  # type: ignore[arg-type]
            execution_results=[], gate_verdicts=[], exergy_improvement_j=0.0, guard_bypass_count=1)

def test_run_returns_one_iteration_per_snapshot(controller: ClosedLoopController, snapshot: SiteSnapshot, now: datetime) -> None:
    iterations = controller.run([snapshot, snapshot, snapshot], at=now)
    assert len(iterations) == 3 and all(isinstance(it, LoopIteration) for it in iterations)

def test_run_all_guard_bypass_zero(controller: ClosedLoopController, snapshot: SiteSnapshot, now: datetime) -> None:
    assert all(it.guard_bypass_count == 0 for it in controller.run([snapshot, snapshot], at=now))

def test_action_log_grows_with_iterations(controller: ClosedLoopController, snapshot: SiteSnapshot, now: datetime) -> None:
    initial = len(controller.action_log)
    controller.step(snapshot, at=now)
    controller.step(snapshot, at=now)
    assert len(controller.action_log) > initial

def test_step_with_no_feasible_decisions(boundary: Boundary, reference: ReferenceState, meta: Metadata, now: datetime) -> None:
    ctrl = ClosedLoopController(LoopConfig(optimizer=ShadowOptimizer(grid_points_per_variable=2),
        policies=[BatteryPolicy()], gate=SafetyGate()))
    battery_state = BatteryState(storage_id="b", stored_energy_j=1_000_000.0, soc=0.5, soh=0.9,
        reserve_energy_j=100_000.0, boundary_id=boundary.boundary_id, reference_state_id=reference.reference_state_id, metadata=meta)
    ts_state = ThermalStorageState(storage_id="ts", layers=[ThermalLayer(temperature_k=320.0, energy_j=1_000_000.0)],
        boundary_id=boundary.boundary_id, reference_state_id=reference.reference_state_id, metadata=meta)
    snap = SiteSnapshot(snapshot_id="minimal-snap", timestamp=now, reference_state=reference, boundary=boundary,
        horizon_duration_s=3600.0, available_pv_power_w=1_000.0, battery_state=battery_state,
        battery_constraints=BatteryConstraints(max_charge_power_w=1000.0, max_discharge_power_w=1000.0,
            min_soc=0.10, max_soc=0.95, round_trip_efficiency=0.9, capacity_j=3_600_000.0*5),
        thermal_storage_state=ts_state,
        thermal_storage_constraints=ThermalStorageConstraints(max_charge_power_w=1000.0, max_discharge_power_w=1000.0,
            charge_efficiency=0.9, discharge_efficiency=0.9, standby_loss_w=10.0),
        building_electric_load_w=800.0, heat_pump_rated_cop=3.0, heat_pump_max_power_w=1000.0,
        delivered_heat_temperature_k=313.15, heat_demand_w=2000.0,
        grid_constraints=GridConstraints(import_limit_w=5000.0, export_limit_w=2000.0,
            import_tariff_per_kwh=0.25, export_tariff_per_kwh=0.05, carbon_intensity_kg_per_kwh=0.4))
    iteration = ctrl.step(snap, at=now)
    assert iteration.guard_bypass_count == 0 and isinstance(iteration, LoopIteration)

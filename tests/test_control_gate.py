"""Tests for SafetyGate — blocked classes, physics checks, batch evaluation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from eie.boundary.boundary import Boundary
from eie.control.action import ActionStatus, ControlAction, ControlTarget
from eie.control.gate import GateVerdict, SafetyGate
from eie.core.enums import BoundaryType, MeasurementMethod
from eie.flows.base import Metadata
from eie.flows.storage import BatteryState, ThermalLayer, ThermalStorageState
from eie.optimization.snapshot import BatteryConstraints, GridConstraints, SiteSnapshot, ThermalStorageConstraints
from eie.reference.state import ReferenceState


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)

@pytest.fixture
def reference(now: datetime) -> ReferenceState:
    return ReferenceState(reference_state_id="ref-gate-test", timestamp=now,
        ambient_temperature_k=293.15, ambient_pressure_pa=101_325.0, confidence=0.99, valid_until=now + timedelta(hours=1))

@pytest.fixture
def boundary(reference: ReferenceState, now: datetime) -> Boundary:
    return Boundary(boundary_id="bnd-gate-test", boundary_type=BoundaryType.SITE,
        included_entity_ids=["battery", "thermal-store"], excluded_entity_ids=[],
        reference_state_id=reference.reference_state_id,
        accounting_period_start=now, accounting_period_end=now + timedelta(hours=1))

@pytest.fixture
def meta(reference: ReferenceState, boundary: Boundary, now: datetime) -> Metadata:
    return Metadata(timestamp=now, source="gate-test", method=MeasurementMethod.SIMULATED,
        unit="J", confidence=1.0, uncertainty=0.0,
        boundary_id=boundary.boundary_id, reference_state_id=reference.reference_state_id)

@pytest.fixture
def snapshot(reference: ReferenceState, boundary: Boundary, meta: Metadata, now: datetime) -> SiteSnapshot:
    battery_state = BatteryState(storage_id="batt-1", stored_energy_j=3_600_000.0*5, soc=0.60, soh=0.95,
        reserve_energy_j=3_600_000.0*0.5, boundary_id=boundary.boundary_id, reference_state_id=reference.reference_state_id, metadata=meta)
    ts_state = ThermalStorageState(storage_id="ts-1", layers=[ThermalLayer(temperature_k=333.15, energy_j=2_000_000.0)],
        boundary_id=boundary.boundary_id, reference_state_id=reference.reference_state_id, metadata=meta)
    return SiteSnapshot(snapshot_id="snap-gate-001", timestamp=now, reference_state=reference, boundary=boundary,
        horizon_duration_s=3600.0, available_pv_power_w=4_000.0, battery_state=battery_state,
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
def gate() -> SafetyGate:
    return SafetyGate()

def _action(action_id: str, target: ControlTarget, value: float, unit: str, safety_class: str,
             risk_level: str = "low", is_reversible: bool = True, now: datetime | None = None) -> ControlAction:
    return ControlAction(action_id=action_id,
        timestamp=now or datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc),
        target=target, target_value=value, unit=unit, source_decision_id="dec-001",
        status=ActionStatus.PENDING, safety_class=safety_class, risk_level=risk_level, is_reversible=is_reversible)

@pytest.mark.parametrize("safety_class,target,value,unit", [
    ("battery", ControlTarget.BATTERY_CHARGE, 500.0, "W"),
    ("thermal", ControlTarget.THERMAL_STORAGE_CHARGE, 500.0, "W"),
    ("load_shift", ControlTarget.LOAD_DEFER, 3600.0, "s"),
    ("waste_heat", ControlTarget.WASTE_HEAT_ROUTE, 340.0, "K"),
])
def test_allowed_classes_approved(gate: SafetyGate, snapshot: SiteSnapshot, safety_class: str, target: ControlTarget, value: float, unit: str) -> None:
    assert gate.evaluate(_action("a1", target, value, unit, safety_class), snapshot).approved

@pytest.mark.parametrize("blocked_class", ["critical_load", "hydrogen", "emergency_reserve", "grid_islanding", "fleet_trade"])
def test_blocked_classes_rejected(gate: SafetyGate, snapshot: SiteSnapshot, blocked_class: str) -> None:
    verdict = gate.evaluate(_action("b1", ControlTarget.BATTERY_CHARGE, 100.0, "W", blocked_class), snapshot)
    assert not verdict.approved
    assert any("blocked" in r.lower() or blocked_class in r for r in verdict.reasons)

def test_unknown_class_rejected(gate: SafetyGate, snapshot: SiteSnapshot) -> None:
    assert not gate.evaluate(_action("u1", ControlTarget.BATTERY_CHARGE, 100.0, "W", "unknown_class"), snapshot).approved

def test_medium_risk_rejected(gate: SafetyGate, snapshot: SiteSnapshot) -> None:
    assert not gate.evaluate(_action("m1", ControlTarget.BATTERY_CHARGE, 100.0, "W", "battery", risk_level="medium"), snapshot).approved

def test_battery_discharge_below_reserve_rejected(gate: SafetyGate, snapshot: SiteSnapshot) -> None:
    verdict = gate.evaluate(_action("d1", ControlTarget.BATTERY_DISCHARGE, 9000.0, "W", "battery"), snapshot)
    assert not verdict.approved
    assert any("reserve" in r.lower() for r in verdict.reasons)

def test_battery_discharge_above_reserve_approved(gate: SafetyGate, snapshot: SiteSnapshot) -> None:
    assert gate.evaluate(_action("d2", ControlTarget.BATTERY_DISCHARGE, 200.0, "W", "battery"), snapshot).approved

def test_battery_charge_exceeds_max_rejected(gate: SafetyGate, snapshot: SiteSnapshot) -> None:
    assert not gate.evaluate(_action("c1", ControlTarget.BATTERY_CHARGE, 5000.0, "W", "battery"), snapshot).approved

def test_thermal_setpoint_below_comfort_rejected(gate: SafetyGate, snapshot: SiteSnapshot) -> None:
    verdict = gate.evaluate(_action("t1", ControlTarget.BUILDING_SETPOINT, 285.0, "K", "thermal"), snapshot)
    assert not verdict.approved
    assert any("comfort" in r.lower() or "below" in r.lower() for r in verdict.reasons)

def test_thermal_setpoint_above_comfort_rejected(gate: SafetyGate, snapshot: SiteSnapshot) -> None:
    assert not gate.evaluate(_action("t2", ControlTarget.BUILDING_SETPOINT, 305.0, "K", "thermal"), snapshot).approved

def test_thermal_setpoint_in_range_approved(gate: SafetyGate, snapshot: SiteSnapshot) -> None:
    assert gate.evaluate(_action("t3", ControlTarget.BUILDING_SETPOINT, 295.0, "K", "thermal"), snapshot).approved

def test_thermal_discharge_exceeds_max_rejected(gate: SafetyGate, snapshot: SiteSnapshot) -> None:
    assert not gate.evaluate(_action("ts1", ControlTarget.THERMAL_STORAGE_DISCHARGE, 5000.0, "W", "thermal"), snapshot).approved

def test_load_shift_irreversible_rejected(gate: SafetyGate, snapshot: SiteSnapshot) -> None:
    assert not gate.evaluate(_action("ls1", ControlTarget.LOAD_DEFER, 3600.0, "s", "load_shift", is_reversible=False), snapshot).approved

def test_load_shift_reversible_approved(gate: SafetyGate, snapshot: SiteSnapshot) -> None:
    assert gate.evaluate(_action("ls2", ControlTarget.LOAD_DEFER, 3600.0, "s", "load_shift", is_reversible=True), snapshot).approved

def test_waste_heat_exceeds_temp_rejected(gate: SafetyGate, snapshot: SiteSnapshot) -> None:
    assert not gate.evaluate(_action("wh1", ControlTarget.WASTE_HEAT_ROUTE, 400.0, "K", "waste_heat"), snapshot).approved

def test_waste_heat_within_bounds_approved(gate: SafetyGate, snapshot: SiteSnapshot) -> None:
    assert gate.evaluate(_action("wh2", ControlTarget.WASTE_HEAT_ROUTE, 340.0, "K", "waste_heat"), snapshot).approved

def test_gate_verdict_frozen(gate: SafetyGate, snapshot: SiteSnapshot) -> None:
    verdict = gate.evaluate(_action("f1", ControlTarget.BATTERY_CHARGE, 100.0, "W", "battery"), snapshot)
    with pytest.raises((AttributeError, TypeError)):
        verdict.approved = False  # type: ignore[misc]

def test_batch_evaluation(gate: SafetyGate, snapshot: SiteSnapshot) -> None:
    actions = [
        _action("b1", ControlTarget.BATTERY_CHARGE, 500.0, "W", "battery"),
        _action("b2", ControlTarget.BATTERY_CHARGE, 100.0, "W", "critical_load"),
        _action("b3", ControlTarget.LOAD_DEFER, 1800.0, "s", "load_shift"),
    ]
    verdicts = gate.evaluate_batch(actions, snapshot)
    assert len(verdicts) == 3
    assert verdicts[0].approved
    assert not verdicts[1].approved
    assert verdicts[2].approved

def test_custom_gate_tighter_reserve(snapshot: SiteSnapshot) -> None:
    custom_gate = SafetyGate(battery_protected_reserve=0.70)
    assert not custom_gate.evaluate(_action("cr1", ControlTarget.BATTERY_DISCHARGE, 200.0, "W", "battery"), snapshot).approved

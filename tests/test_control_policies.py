"""Tests for all five ControlPolicy implementations."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from eie.boundary.boundary import Boundary
from eie.control.action import ActionStatus, ControlTarget
from eie.control.policy import BatteryPolicy, LoadShiftPolicy, PreCoolingPolicy, ThermalStoragePolicy, WasteHeatPolicy
from eie.core.enums import BoundaryType, MeasurementMethod
from eie.flows.base import Metadata
from eie.flows.storage import BatteryState, ThermalLayer, ThermalStorageState
from eie.optimization.decision import DispatchDecision
from eie.optimization.snapshot import BatteryConstraints, GridConstraints, SiteSnapshot, ThermalStorageConstraints
from eie.reference.state import ReferenceState


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)

@pytest.fixture
def reference(now: datetime) -> ReferenceState:
    return ReferenceState(reference_state_id="ref-policy-test", timestamp=now,
        ambient_temperature_k=293.15, ambient_pressure_pa=101_325.0, confidence=0.99, valid_until=now + timedelta(hours=1))

@pytest.fixture
def boundary(reference: ReferenceState, now: datetime) -> Boundary:
    return Boundary(boundary_id="bnd-policy-test", boundary_type=BoundaryType.SITE,
        included_entity_ids=["battery", "thermal-store"], excluded_entity_ids=[],
        reference_state_id=reference.reference_state_id,
        accounting_period_start=now, accounting_period_end=now + timedelta(hours=1))

@pytest.fixture
def meta(reference: ReferenceState, boundary: Boundary, now: datetime) -> Metadata:
    return Metadata(timestamp=now, source="policy-test", method=MeasurementMethod.SIMULATED, unit="J",
        confidence=1.0, uncertainty=0.0, boundary_id=boundary.boundary_id, reference_state_id=reference.reference_state_id)

@pytest.fixture
def snapshot(reference: ReferenceState, boundary: Boundary, meta: Metadata, now: datetime) -> SiteSnapshot:
    battery_state = BatteryState(storage_id="batt-1", stored_energy_j=3_600_000.0*5, soc=0.60, soh=0.95,
        reserve_energy_j=3_600_000.0*0.5, boundary_id=boundary.boundary_id, reference_state_id=reference.reference_state_id, metadata=meta)
    ts_state = ThermalStorageState(storage_id="ts-1", layers=[ThermalLayer(temperature_k=333.15, energy_j=2_000_000.0)],
        boundary_id=boundary.boundary_id, reference_state_id=reference.reference_state_id, metadata=meta)
    return SiteSnapshot(snapshot_id="snap-policy-001", timestamp=now, reference_state=reference, boundary=boundary,
        horizon_duration_s=3600.0, available_pv_power_w=5_000.0, battery_state=battery_state,
        battery_constraints=BatteryConstraints(max_charge_power_w=3_000.0, max_discharge_power_w=3_000.0,
            min_soc=0.10, max_soc=0.95, round_trip_efficiency=0.92, capacity_j=3_600_000.0*10.0),
        thermal_storage_state=ts_state,
        thermal_storage_constraints=ThermalStorageConstraints(max_charge_power_w=2_000.0, max_discharge_power_w=2_000.0,
            charge_efficiency=0.95, discharge_efficiency=0.95, standby_loss_w=50.0),
        building_electric_load_w=1_200.0, heat_pump_rated_cop=3.5, heat_pump_max_power_w=2_000.0,
        delivered_heat_temperature_k=318.15, heat_demand_w=3_000.0,
        grid_constraints=GridConstraints(import_limit_w=10_000.0, export_limit_w=5_000.0,
            import_tariff_per_kwh=0.28, export_tariff_per_kwh=0.05, carbon_intensity_kg_per_kwh=0.42))

def _decision(battery_charge: float, thermal_storage: float, hp: float, boundary: Boundary, reference: ReferenceState, now: datetime) -> DispatchDecision:
    from eie.guards.physics_guard import GuardResult
    from eie.core.enums import Severity
    guard = GuardResult(guard_name="PhysicsGuard", passed=True, severity=Severity.INFO, reason="ok", evidence={})
    return DispatchDecision(decision_id="dec-policy-test", timestamp=now, horizon_start=now,
        horizon_end=now + timedelta(hours=1), boundary_id=boundary.boundary_id, reference_state_id=reference.reference_state_id,
        variable_values={"battery_charge_fraction": battery_charge, "thermal_storage_fraction": thermal_storage, "heat_pump_fraction": hp},
        is_advisory=True, guard_verified=True, guard_results=[guard], feasibility_notes=[], metadata={})

def test_battery_policy_charge(snapshot: SiteSnapshot, boundary: Boundary, reference: ReferenceState, now: datetime) -> None:
    actions = BatteryPolicy(protected_reserve_fraction=0.20).actions_from_decision(_decision(0.8, 0.5, 0.5, boundary, reference, now), snapshot, now)
    assert len(actions) == 1 and actions[0].target == ControlTarget.BATTERY_CHARGE and actions[0].target_value > 0

def test_battery_policy_discharge(snapshot: SiteSnapshot, boundary: Boundary, reference: ReferenceState, now: datetime) -> None:
    actions = BatteryPolicy(protected_reserve_fraction=0.10).actions_from_decision(_decision(0.1, 0.5, 0.5, boundary, reference, now), snapshot, now)
    assert len(actions) == 1 and actions[0].target == ControlTarget.BATTERY_DISCHARGE and actions[0].target_value > 0

def test_battery_policy_reserve_clamps_discharge(snapshot: SiteSnapshot, boundary: Boundary, reference: ReferenceState, now: datetime) -> None:
    assert len(BatteryPolicy(protected_reserve_fraction=0.60).actions_from_decision(_decision(0.0, 0.5, 0.5, boundary, reference, now), snapshot, now)) == 0

def test_battery_policy_invalid_reserve() -> None:
    with pytest.raises(ValueError):
        BatteryPolicy(protected_reserve_fraction=1.5)

def test_battery_policy_all_actions_pending(snapshot: SiteSnapshot, boundary: Boundary, reference: ReferenceState, now: datetime) -> None:
    actions = BatteryPolicy().actions_from_decision(_decision(0.9, 0.5, 0.5, boundary, reference, now), snapshot, now)
    for a in actions:
        assert a.status == ActionStatus.PENDING and a.risk_level == "low"

def test_thermal_policy_charge(snapshot: SiteSnapshot, boundary: Boundary, reference: ReferenceState, now: datetime) -> None:
    actions = ThermalStoragePolicy(min_comfort_k=293.15, max_comfort_k=333.15).actions_from_decision(_decision(0.5, 0.9, 0.5, boundary, reference, now), snapshot, now)
    assert len(actions) == 1 and actions[0].target == ControlTarget.THERMAL_STORAGE_CHARGE and actions[0].safety_class == "thermal"

def test_thermal_policy_discharge(snapshot: SiteSnapshot, boundary: Boundary, reference: ReferenceState, now: datetime) -> None:
    actions = ThermalStoragePolicy(min_comfort_k=293.15, max_comfort_k=333.15).actions_from_decision(_decision(0.5, 0.1, 0.5, boundary, reference, now), snapshot, now)
    assert len(actions) == 1 and actions[0].target == ControlTarget.THERMAL_STORAGE_DISCHARGE

def test_thermal_policy_neutral(snapshot: SiteSnapshot, boundary: Boundary, reference: ReferenceState, now: datetime) -> None:
    assert len(ThermalStoragePolicy(min_comfort_k=293.15, max_comfort_k=333.15).actions_from_decision(_decision(0.5, 0.5, 0.5, boundary, reference, now), snapshot, now)) == 0

def test_thermal_policy_invalid_bounds() -> None:
    with pytest.raises(ValueError):
        ThermalStoragePolicy(min_comfort_k=340.0, max_comfort_k=320.0)

def test_pre_cooling_activates_above_threshold(snapshot: SiteSnapshot, boundary: Boundary, reference: ReferenceState, now: datetime) -> None:
    actions = PreCoolingPolicy(comfort_lower_k=291.15, comfort_upper_k=299.15, activation_threshold=0.3).actions_from_decision(_decision(0.5, 0.5, 0.8, boundary, reference, now), snapshot, now)
    assert len(actions) == 1 and actions[0].target == ControlTarget.PRE_COOL_SETPOINT
    assert actions[0].unit == "K" and 291.15 <= actions[0].target_value <= 299.15

def test_pre_cooling_no_action_below_threshold(snapshot: SiteSnapshot, boundary: Boundary, reference: ReferenceState, now: datetime) -> None:
    assert len(PreCoolingPolicy(comfort_lower_k=291.15, comfort_upper_k=299.15, activation_threshold=0.3).actions_from_decision(_decision(0.5, 0.5, 0.1, boundary, reference, now), snapshot, now)) == 0

def test_pre_cooling_setpoint_clamped(snapshot: SiteSnapshot, boundary: Boundary, reference: ReferenceState, now: datetime) -> None:
    actions = PreCoolingPolicy(comfort_lower_k=291.15, comfort_upper_k=299.15).actions_from_decision(_decision(0.5, 0.5, 1.0, boundary, reference, now), snapshot, now)
    assert len(actions) == 1 and 291.15 <= actions[0].target_value <= 299.15

def test_load_shift_activates_when_pv_low(snapshot: SiteSnapshot, boundary: Boundary, reference: ReferenceState, now: datetime) -> None:
    assert len(LoadShiftPolicy(["hvac", "ev-charger"], max_defer_s=3600.0).actions_from_decision(_decision(0.5, 0.5, 0.5, boundary, reference, now), snapshot, now)) == 0

def test_load_shift_activates_low_pv(boundary: Boundary, reference: ReferenceState, meta: Metadata, now: datetime) -> None:
    battery_state = BatteryState(storage_id="b", stored_energy_j=1_000_000.0, soc=0.5, soh=0.95,
        reserve_energy_j=100_000.0, boundary_id=boundary.boundary_id, reference_state_id=reference.reference_state_id, metadata=meta)
    ts_state = ThermalStorageState(storage_id="ts", layers=[ThermalLayer(temperature_k=320.0, energy_j=500_000.0)],
        boundary_id=boundary.boundary_id, reference_state_id=reference.reference_state_id, metadata=meta)
    low_pv_snapshot = SiteSnapshot(snapshot_id="low-pv", timestamp=now, reference_state=reference, boundary=boundary,
        horizon_duration_s=3600.0, available_pv_power_w=100.0, battery_state=battery_state,
        battery_constraints=BatteryConstraints(max_charge_power_w=3000.0, max_discharge_power_w=3000.0,
            min_soc=0.10, max_soc=0.95, round_trip_efficiency=0.92, capacity_j=3_600_000.0*10.0),
        thermal_storage_state=ts_state,
        thermal_storage_constraints=ThermalStorageConstraints(max_charge_power_w=2000.0, max_discharge_power_w=2000.0,
            charge_efficiency=0.95, discharge_efficiency=0.95, standby_loss_w=50.0),
        building_electric_load_w=3_000.0, heat_pump_rated_cop=3.5, heat_pump_max_power_w=2000.0,
        delivered_heat_temperature_k=318.15, heat_demand_w=2000.0,
        grid_constraints=GridConstraints(import_limit_w=10000.0, export_limit_w=5000.0,
            import_tariff_per_kwh=0.28, export_tariff_per_kwh=0.05, carbon_intensity_kg_per_kwh=0.42))
    from eie.guards.physics_guard import GuardResult
    from eie.core.enums import Severity
    guard = GuardResult("PhysicsGuard", True, Severity.INFO, "ok", {})
    decision = DispatchDecision(decision_id="d-ls", timestamp=now, horizon_start=now, horizon_end=now + timedelta(hours=1),
        boundary_id=boundary.boundary_id, reference_state_id=reference.reference_state_id,
        variable_values={"battery_charge_fraction": 0.5, "thermal_storage_fraction": 0.5, "heat_pump_fraction": 0.5},
        is_advisory=True, guard_verified=True, guard_results=[guard], feasibility_notes=[], metadata={})
    actions = LoadShiftPolicy(["hvac", "ev-charger"], max_defer_s=3600.0, activation_pv_threshold=0.5).actions_from_decision(decision, low_pv_snapshot, now)
    assert len(actions) == 2
    for a in actions:
        assert a.is_reversible and a.safety_class == "load_shift" and a.target == ControlTarget.LOAD_DEFER

def test_load_shift_empty_load_ids() -> None:
    with pytest.raises(ValueError):
        LoadShiftPolicy([])

def test_waste_heat_activates_high_hp(snapshot: SiteSnapshot, boundary: Boundary, reference: ReferenceState, now: datetime) -> None:
    actions = WasteHeatPolicy(max_heat_rate_w=5000.0, safe_temperature_bounds_k=(320.0, 360.0)).actions_from_decision(_decision(0.5, 0.5, 0.9, boundary, reference, now), snapshot, now)
    assert len(actions) == 1 and actions[0].target == ControlTarget.WASTE_HEAT_ROUTE
    assert actions[0].safety_class == "waste_heat" and actions[0].unit == "K"
    assert pytest.approx(actions[0].target_value) == 340.0

def test_waste_heat_no_action_low_hp(snapshot: SiteSnapshot, boundary: Boundary, reference: ReferenceState, now: datetime) -> None:
    assert len(WasteHeatPolicy(max_heat_rate_w=5000.0, safe_temperature_bounds_k=(320.0, 360.0)).actions_from_decision(_decision(0.5, 0.5, 0.3, boundary, reference, now), snapshot, now)) == 0

def test_waste_heat_invalid_rate() -> None:
    with pytest.raises(ValueError):
        WasteHeatPolicy(max_heat_rate_w=0.0, safe_temperature_bounds_k=(320.0, 360.0))

def test_waste_heat_invalid_bounds() -> None:
    with pytest.raises(ValueError):
        WasteHeatPolicy(max_heat_rate_w=1000.0, safe_temperature_bounds_k=(360.0, 320.0))
